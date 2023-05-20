import java.util.*;
import java.io.*;
import java.lang.*;
public class bracelet
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("bracelet.in"));
      char[] a = in.next().toCharArray();
      if(a.length % 2 != 0)
      {
         System.out.println(0);
         System.exit(0);
      }
      
      int count = 0;
      for(char c : a)
      {
         if(c == '(') count++;
         else count--;
      }
      
      int ans = 0;
      int val = 0;
      if(count == -2)
      {
         for(int i = 0; i < a.length; i++)
         {
            if(a[i] == '(') val++;
            else
            {
               val--;
               if(i < a.length - 1)
                  ans++;
               if(val < 0)
               {
                  System.out.println(ans);
                  System.exit(0);
               }
            }
         }      
      }
      else if(count == 2)
      {
         for(int i = a.length - 1; i >= 0; i--)
         {
            if(a[i] == ')') val++;
            else
            {
               val--;
               if(i > 0)
                  ans++;
               if(val < 0)
               {
                  System.out.println(ans);
                  System.exit(0);
               }
            }
         }      
      }
      else
      {
         System.out.println(0);
         System.exit(0);
      }
   }
}